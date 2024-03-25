; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs -amdgpu-codegenprepare-break-large-phis=false < %s | FileCheck %s

; CHECK-LABEL: {{^}}_amdgpu_ps_main:
;
; This test case used to hit an assertion in SIOptimizeVGPRLiveRange because of a vector with dead lanes
; leading to an effectively dead INSERT_SUBREG.


define dllexport amdgpu_ps void @_amdgpu_ps_main(i32 %descTable2) #0 {
.entry:
  %i2 = zext i32 %descTable2 to i64
  %i4 = inttoptr i64 %i2 to ptr addrspace(4)
  %i159 = call reassoc nnan nsz arcp contract afn <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 15, float poison, float poison, float poison, float poison, float poison, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %i1540 = shufflevector <4 x float> %i159, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %i1526 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> poison, i32 2688, i32 0)
  %i1746 = load <4 x i32>, ptr addrspace(4) %i4, align 16
  br label %bb1750

bb1750:                                           ; preds = %bb1897, %.entry
  %__llpc_global_proxy_r3.12.vec.extract2358295 = phi i32 [ 0, %.entry ], [ %__llpc_global_proxy_r3.12.vec.extract2358, %bb1897 ]
  %__llpc_global_proxy_r13.20293 = phi <4 x i32> [ undef, %.entry ], [ %__llpc_global_proxy_r13.22, %bb1897 ]
  %__llpc_global_proxy_r10.19291 = phi <4 x i32> [ poison, %.entry ], [ %i1914, %bb1897 ]
  %i1751 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %i1746, i32 poison, i32 0, i32 0, i32 0)
  %i1754 = shufflevector <4 x i32> %__llpc_global_proxy_r10.19291, <4 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %__llpc_global_proxy_r7.12.vec.extract1953260 = bitcast float %i1751 to i32
  %i1760 = or i32 %__llpc_global_proxy_r7.12.vec.extract1953260, %__llpc_global_proxy_r3.12.vec.extract2358295
  %i1786 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> poison, i32 %i1760, i32 0)
  %__llpc_global_proxy_r11.12.vec.insert1237 = insertelement <4 x i32> %i1754, i32 %i1786, i64 3
  %.not2783 = icmp eq i32 %i1786, 0
  br i1 %.not2783, label %bb1789, label %bb1787

bb1787:                                           ; preds = %bb1750
  %i1788 = shufflevector <4 x i32> <i32 0, i32 0, i32 0, i32 poison>, <4 x i32> %__llpc_global_proxy_r13.20293, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  br label %bb1897

bb1789:                                           ; preds = %bb1750
  %i1796 = call reassoc nnan nsz arcp contract afn float @llvm.amdgcn.fmed3.f32(float poison, float 0.000000e+00, float 1.000000e+00)
  %i1797 = fmul reassoc nnan nsz arcp contract afn float %i1796, %i1796
  %i1800 = fdiv reassoc nnan nsz arcp contract afn float %i1797, 0.000000e+00
  %i1801 = bitcast float %i1800 to i32
  %__llpc_global_proxy_r11.12.vec.insert1245 = insertelement <4 x i32> %__llpc_global_proxy_r11.12.vec.insert1237, i32 poison, i64 3
  %i1818 = call reassoc nnan nsz arcp contract afn float @llvm.amdgcn.fmed3.f32(float poison, float 0.000000e+00, float 1.000000e+00)
  %i1819 = bitcast float %i1818 to i32
  %i1878 = shufflevector <4 x i32> %__llpc_global_proxy_r11.12.vec.insert1245, <4 x i32> poison, <3 x i32> <i32 3, i32 3, i32 3>
  %i1879 = bitcast <3 x i32> %i1878 to <3 x float>
  %i1881 = fmul reassoc nnan nsz arcp contract afn <3 x float> %i1540, %i1879
  %i1882 = call <3 x i32> @llvm.amdgcn.s.buffer.load.v3i32(<4 x i32> poison, i32 poison, i32 0)
  %i1883 = shufflevector <3 x i32> %i1882, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %i1884 = bitcast <4 x i32> %i1883 to <4 x float>
  %i1885 = shufflevector <4 x float> %i1884, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %i1886 = insertelement <3 x i32> undef, i32 %i1819, i64 0
  %i1887 = bitcast <3 x i32> %i1886 to <3 x float>
  %i1888 = insertelement <3 x i32> undef, i32 %i1801, i64 0
  %i1889 = bitcast <3 x i32> %i1888 to <3 x float>
  %i1890 = fmul reassoc nnan nsz arcp contract afn <3 x float> %i1887, %i1889
  %i1891 = shufflevector <3 x float> %i1890, <3 x float> poison, <3 x i32> zeroinitializer
  %i1892 = fmul reassoc nnan nsz arcp contract afn <3 x float> %i1885, %i1891
  %i1893 = fmul reassoc nnan nsz arcp contract afn <3 x float> %i1892, %i1881
  %i1894 = bitcast <3 x float> %i1893 to <3 x i32>
  %i1895 = shufflevector <3 x i32> %i1894, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %i1896 = insertelement <4 x i32> %i1895, i32 %i1819, i64 3
  br label %bb1897

bb1897:                                           ; preds = %bb1789, %bb1787
  %__llpc_global_proxy_r11.19 = phi <4 x i32> [ %__llpc_global_proxy_r11.12.vec.insert1237, %bb1787 ], [ %__llpc_global_proxy_r11.12.vec.insert1245, %bb1789 ]
  %__llpc_global_proxy_r13.22 = phi <4 x i32> [ %i1788, %bb1787 ], [ %i1896, %bb1789 ]
  %i1898 = shufflevector <4 x i32> %__llpc_global_proxy_r11.19, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %i1899 = bitcast <3 x i32> %i1898 to <3 x float>
  %i1900 = shufflevector <4 x i32> %__llpc_global_proxy_r13.22, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %i1901 = bitcast <3 x i32> %i1900 to <3 x float>
  %i1902 = fadd reassoc nnan nsz arcp contract afn <3 x float> %i1901, %i1899
  %i1903 = bitcast <3 x float> %i1902 to <3 x i32>
  %i1907 = shufflevector <3 x i32> %i1903, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %i1908 = shufflevector <4 x i32> %i1907, <4 x i32> %__llpc_global_proxy_r11.19, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %i1914 = shufflevector <4 x i32> %i1908, <4 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %__llpc_global_proxy_r3.12.vec.extract2358 = extractelement <2 x i32> zeroinitializer, i64 1
  %.not2780.not = icmp ult i32 %__llpc_global_proxy_r3.12.vec.extract2358, %i1526
  br i1 %.not2780.not, label %bb1750, label %._crit_edge298

._crit_edge298:                                   ; preds = %bb1897
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f32.f32(i32 immarg, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)
declare float @llvm.amdgcn.fmed3.f32(float, float, float)
declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i32 immarg)
declare <3 x i32> @llvm.amdgcn.s.buffer.load.v3i32(<4 x i32>, i32, i32 immarg)

attributes #0 = { "target-features"=",+wavefrontsize64,+cumode" }
