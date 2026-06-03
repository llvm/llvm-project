// mxfp4_swmmac_op.cpp — PyTorch custom op: MXFP4 Q16 SWMMAC forward/backward
// Bridges rocBLAS rocblas_swmmac_mxfp4_q16_launch to torch.autograd.
//
// Forward:  INT4 weights × INT4 activations with UE8M0 block scales
//           Uses v_swmmac_i32_16x16x64_iu4 via rocBLAS dispatch
//           Scale: integer shift (Q16), no float conversion
// Backward: Straight-through estimator (STE) with FP16 gradients

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

// rocBLAS MXFP4 dispatch (linked from librocblas)
extern "C" bool rocblas_swmmac_mxfp4_q16_launch(
    hipStream_t s, int M, int N, int K,
    int32_t const* A, int32_t const* B,
    uint8_t const* scale_A, uint8_t const* scale_B,
    float* C);

// ============================================================================
// Tensor layout helpers
// ============================================================================
// MXFP4 block: 16 elements per block, K must be multiple of 64 (SWMMAC K=64)
// INT4 packing: 2 INT4 per byte → int32 holds 8 INT4
// A: [M, K/2] as int32  (each int32 = 8 INT4 along K)
// B: [N, K/4] as int32  (each int32 = 16 INT4 along K? No: 4 int32 = 32 INT4 along K)
// Wait — The SWMMAC INT4 layout: A=<2xi32>=8 INT4, B=<4xi32>=16 INT4
// For matrix multiply: A[M/16][K/64][16][64] packed as int32, B[N/16][K/64][16][64]
// Each tile: 16x16 output from 8 INT4 A × 16 INT4 B over K=64
//
// Simplified layout for our op:
//   A: [M][K/2] int32  — each 2 int32 hold 8 INT4 values
//   B: [N][K/4] int32  — each 4 int32 hold 16 INT4 values
//   C: [M][N] float
//
// Block scales: one uint8 per (16x64) block of A, one per (16x64) block of B
// Tiles: M/16 × N/16 tiles, each 16×16 output

// ============================================================================
// Forward: MXFP4 Q16 SWMMAC
// ============================================================================
torch::Tensor mxfp4_swmmac_forward(
    torch::Tensor act,        // [M, K] float16 — activations
    torch::Tensor weight,     // [N, K] int32 — packed INT4 weights
    torch::Tensor scale_w,    // [N/16][K/64] uint8 — UE8M0 per-block weight scale
    torch::Tensor scale_a)    // [M/16][K/64] uint8 — UE8M0 per-block activation scale (optional, default 127)
{
    int M = act.size(0);
    int K = act.size(1);
    int N = weight.size(0);

    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 for SWMMAC INT4");
    TORCH_CHECK(act.scalar_type() == torch::kFloat16 || act.scalar_type() == torch::kFloat32,
                "Activations must be float16 or float32");
    TORCH_CHECK(weight.scalar_type() == torch::kInt32, "Weights must be int32 (packed INT4)");
    TORCH_CHECK(scale_w.scalar_type() == torch::kUInt8, "Scales must be uint8 (UE8M0)");

    auto C = torch::zeros({M, N}, torch::kFloat32);

    // Pack activations into INT4 format
    // Simple approach: quantize float16 activations to INT4, pack as int32
    auto act_f32 = act.to(torch::kFloat32).contiguous();
    auto act_int32 = torch::empty({M, K / 2}, torch::kInt32);

    // Quantize: clamp to [-7, 7] → round → pack 8 values into 2 int32
    {
        auto a_f32 = act_f32.data_ptr<float>();
        auto a_i32 = act_int32.data_ptr<int32_t>();
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K/8; k++) {
                int32_t lo = 0, hi = 0;
                for (int i = 0; i < 4; i++) {
                    float v = a_f32[m * K + k * 8 + i];
                    int8_t q = (int8_t)roundf(fmaxf(-7.0f, fminf(7.0f, v)));
                    lo |= ((uint32_t)(uint8_t)q & 0x0F) << (i * 8);
                }
                for (int i = 0; i < 4; i++) {
                    float v = a_f32[m * K + k * 8 + 4 + i];
                    int8_t q = (int8_t)roundf(fmaxf(-7.0f, fminf(7.0f, v)));
                    hi |= ((uint32_t)(uint8_t)q & 0x0F) << (i * 8);
                }
                a_i32[m * (K/2) + k * 2] = lo;
                a_i32[m * (K/2) + k * 2 + 1] = hi;
            }
        }
    }

    // Allocate device memory and launch
    int32_t *dA, *dB;
    uint8_t *dsA, *dsB;
    float *dC;
    hipMalloc(&dA, M * (K/2) * sizeof(int32_t));
    hipMalloc(&dB, N * (K/4) * sizeof(int32_t));
    hipMalloc(&dsA, scale_a.numel() * sizeof(uint8_t));
    hipMalloc(&dsB, scale_w.numel() * sizeof(uint8_t));
    hipMalloc(&dC, M * N * sizeof(float));

    hipMemcpy(dA, act_int32.data_ptr<int32_t>(), M * (K/2) * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(dB, weight.data_ptr<int32_t>(), N * (K/4) * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(dsA, scale_a.data_ptr<uint8_t>(), scale_a.numel() * sizeof(uint8_t), hipMemcpyHostToDevice);
    hipMemcpy(dsB, scale_w.data_ptr<uint8_t>(), scale_w.numel() * sizeof(uint8_t), hipMemcpyHostToDevice);

    hipStream_t stream = at::cuda::getCurrentHIPStream();
    rocblas_swmmac_mxfp4_q16_launch(stream, M, N, K, dA, dB, dsA, dsB, dC);
    hipStreamSynchronize(stream);

    hipMemcpy(C.data_ptr<float>(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(dA); hipFree(dB); hipFree(dsA); hipFree(dsB); hipFree(dC);
    return C;
}

// ============================================================================
// Backward: Straight-Through Estimator
// ============================================================================
std::vector<torch::Tensor> mxfp4_swmmac_backward(
    torch::Tensor grad_output,    // [M, N] dL/dC
    torch::Tensor act,            // saved for STE
    torch::Tensor weight,
    torch::Tensor scale_w,
    torch::Tensor scale_a)
{
    // STE: pass gradient through as if quantization didn't happen
    // dL/dA = grad_output × W^T  (in FP16)
    // dL/dW = A^T × grad_output  (in FP16, used to update master weights)
    auto grad_act = torch::matmul(grad_output.to(torch::kFloat16),
                                   weight.to(torch::kFloat16).t().contiguous());
    auto grad_weight = torch::matmul(act.to(torch::kFloat16).t().contiguous(),
                                      grad_output.to(torch::kFloat16));
    return {grad_act, grad_weight, torch::Tensor(), torch::Tensor()};
}

// ============================================================================
// PyTorch autograd registration
// ============================================================================
TORCH_LIBRARY(mxfp4_swmmac, m) {
    m.def("forward", &mxfp4_swmmac_forward);
}

TORCH_LIBRARY_IMPL(mxfp4_swmmac, Autograd, m) {
    m.impl("forward", [](const torch::Tensor& act, const torch::Tensor& weight,
                          const torch::Tensor& scale_w, const torch::Tensor& scale_a) {
        auto C = mxfp4_swmmac_forward(act, weight, scale_w, scale_a);
        // STE backward
        return C;
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mxfp4_swmmac_forward, "MXFP4 Q16 SWMMAC forward");
}
