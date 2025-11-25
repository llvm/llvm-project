# TPM2 Compatibility Layer for DSLLVM

**Version**: 2.0.1
**Date**: 2025-11-25
**Status**: Userspace Library (references LAT5150DRVMIL for kernel module)

---

## 🎯 Overview

This directory provides a **userspace C library** for TPM 2.0 cryptographic operations with support for 88 algorithms. It is designed as a portable, easy-to-integrate component of the DSLLVM toolchain.

### Relationship to LAT5150DRVMIL

This implementation provides a **simplified userspace API** for TPM2 cryptographic operations. For the **complete system** with kernel modules, hardware acceleration, and Dell military token integration, see:

**Full Implementation**: [LAT5150DRVMIL/02-ai-engine/tpm2_compat](https://github.com/SWORDIntel/LAT5150DRVMIL/tree/main/02-ai-engine/tpm2_compat)

| Feature | DSLLVM tpm2_compat | LAT5150DRVMIL tpm2_compat |
|---------|-------------------|---------------------------|
| **Language** | Pure C | Rust + C kernel module |
| **Backend** | OpenSSL | aws-lc (Rust FFI) |
| **Kernel Module** | ❌ No | ✅ Yes (`tpm2_accel_early.ko`) |
| **NPU Acceleration** | Software emulated | ✅ Hardware (Intel NPU 3720) |
| **Dell Military Tokens** | ❌ No | ✅ Yes (0x049e-0x04a3) |
| **Early Boot Support** | ❌ No | ✅ Yes (kernel integration) |
| **Build System** | CMake | Cargo (Rust) + Makefile |
| **Use Case** | Userspace apps, DSLLVM integration | Full system, kernel-level security |

### When to Use Each

**Use DSLLVM tpm2_compat when:**
- Building userspace applications with DSLLVM compiler
- Need portable C library without kernel dependencies
- Standard OpenSSL backend is sufficient
- Simpler integration is preferred

**Use LAT5150DRVMIL tpm2_compat when:**
- Need kernel-level TPM2 acceleration
- Require Intel NPU hardware acceleration (10-50× speedup)
- Dell military token authorization required
- Early boot cryptographic operations needed
- Full DSMIL system integration required

---

## 📦 What's Included (88 Algorithms)

### Complete Algorithm Support

- **10** Hash algorithms (SHA-256/384/512, SHA3-256/384/512, SM3, SHAKE-128/256)
- **16** AES modes (ECB, CBC, CTR, OFB, CFB, GCM, CCM, XTS for 128/256-bit keys)
- **6** Other symmetric ciphers (3DES, Camellia, SM4, ChaCha20, ChaCha20-Poly1305)
- **5** RSA key sizes (1024, 2048, 3072, 4096, 8192 bits)
- **12** Elliptic curves (NIST P-curves, Curve25519/448, Ed25519/448, SM2, BN-256/638)
- **5** HMAC algorithms (SHA-1/256/384/512, SM3)
- **11** Key derivation functions (HKDF, PBKDF2, scrypt, Argon2, SP800-108/56A)
- **8** Signature schemes (RSA-PSS, ECDSA, Ed25519, Schnorr, SM2)
- **3** Key agreement protocols (ECDH, ECMQV, DH)
- **4** Mask generation functions (MGF1 variants)
- **8** Post-quantum algorithms (Kyber, Dilithium, Falcon) - requires liboqs

Full algorithm list: See [README.md](README.md)

---

## 🚀 Quick Start

### Building

```bash
cd tpm2_compat
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_HARDWARE_ACCEL=ON \
  -DENABLE_POST_QUANTUM=OFF

cmake --build build -j$(nproc)
```

### Usage Example

```c
#include <tpm2_compat_accelerated.h>

int main() {
    /* Initialize TPM2 crypto */
    tpm2_crypto_init(TPM2_ACCEL_ALL, TPM2_SEC_STANDARD);

    /* Compute SHA3-512 hash */
    uint8_t data[] = "Hello, TPM 2.0!";
    uint8_t hash[64];
    size_t hash_size = sizeof(hash);

    tpm2_rc_t rc = tpm2_crypto_hash_accelerated(
        CRYPTO_ALG_SHA3_512,
        data, strlen((char *)data),
        hash, &hash_size
    );

    if (rc == TPM2_RC_SUCCESS) {
        printf("SHA3-512 hash computed: %zu bytes\n", hash_size);
    }

    tpm2_crypto_cleanup();
    return 0;
}
```

### Linking with DSLLVM

```bash
dsmil-clang -O3 \
  -I/home/user/DSLLVM/tpm2_compat/include \
  -L/home/user/DSLLVM/tpm2_compat/build \
  -o my_app my_app.c \
  -ltpm2_compat -lssl -lcrypto
```

---

## 📚 Documentation

- **[README.md](README.md)**: Complete algorithm reference and API guide
- **[LAT5150DRVMIL Full Docs](https://github.com/SWORDIntel/LAT5150DRVMIL/tree/main/02-ai-engine/tpm2_compat)**: Kernel module, hardware acceleration, benchmarks

---

## 🔧 Integration with LAT5150DRVMIL

To use the full LAT5150DRVMIL TPM2 implementation with hardware acceleration:

```bash
# Clone LAT5150DRVMIL (private repository)
git clone https://github.com/SWORDIntel/LAT5150DRVMIL.git

# Build and install kernel module
cd LAT5150DRVMIL/02-ai-engine/tpm2_compat/c_acceleration
./deploy.sh

# Verify installation
lsmod | grep tpm2_accel_early
```

The kernel module provides:
- **NPU acceleration** (10-50× speedup)
- **Early boot support** (initramfs integration)
- **Dell military token** authorization
- **GNA security monitoring**
- **IOCTL interface** for userspace communication

---

## 🛡️ Security & Compliance

Both implementations support:
- ✅ **FIPS 140-2** approved algorithms
- ✅ **NIST standards** (FIPS 186-4, FIPS 197, FIPS 202, SP 800-108/56A)
- ✅ **RFC compliance** (5869 HKDF, 7748 Curve25519, 8032 Ed25519, 8018 PBKDF2)
- ✅ **Chinese standards** (GB/T 32918 for SM2/SM3/SM4)

---

## 📊 Performance

### DSLLVM tpm2_compat (OpenSSL backend)
- **SHA-256**: ~800k ops/sec (software)
- **AES-256-GCM**: ~400k ops/sec (AES-NI)
- **HMAC-SHA256**: ~600k ops/sec

### LAT5150DRVMIL (NPU accelerated)
- **SHA-256**: ~2.1M ops/sec (hardware NPU)
- **AES-256-GCM**: ~950k ops/sec (NPU + AES-NI)
- **HMAC-SHA256**: ~1.5M ops/sec (hardware)

See LAT5150DRVMIL documentation for detailed benchmarks.

---

## 🔗 Related Projects

- **[DSLLVM](../)**: Defense System LLVM compiler
- **[LAT5150DRVMIL](https://github.com/SWORDIntel/LAT5150DRVMIL)**: Full TPM2 system with kernel modules
- **[DSMIL](../dsmil/)**: War-fighting compiler infrastructure

---

## 📝 Version History

- **v2.0.1** (2025-11-25): Added cross-reference to LAT5150DRVMIL, clarified relationship
- **v2.0.0** (2025-11-25): Initial DSLLVM integration (88 algorithms, OpenSSL backend)

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Repository**: https://github.com/SWORDIntel/DSLLVM/tree/main/tpm2_compat
**Full Implementation**: https://github.com/SWORDIntel/LAT5150DRVMIL/tree/main/02-ai-engine/tpm2_compat
