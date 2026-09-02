; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Add S2_pstorerft_io and S2_pstorerff_io to isValidOffset.
; Ensure that llc compiles this test cleanly and emits the predicated .h stores.

; CHECK-LABEL: pr196966:
; CHECK: if (!{{p[0-9]+(\.new)?}}) memh({{r[0-9]+}}+#{{[0-9]+}}) = {{r[0-9]+}}.h

declare <16 x i16> @helper()
declare void @llvm.masked.scatter.v16i16.v16p0(<16 x i16>, <16 x ptr>, <16 x i1>)

define void @pr196966(ptr %base) {
entry:
  %mb = icmp ne ptr %base, null
  br i1 %mb, label %then, label %common.ret

common.ret:
  ret void

then:
  %p4 = getelementptr i8, ptr %base, i64 20
  %call = call <16 x i16> @helper()
  %pv4 = insertelement <16 x ptr> poison, ptr %p4, i64 4
  %ptrs.raw = freeze <16 x ptr> %pv4
  %p2 = getelementptr i8, ptr %base, i64 6
  %pv2 = insertelement <16 x ptr> poison, ptr %p2, i64 2
  %ptrs = shufflevector <16 x ptr> %ptrs.raw, <16 x ptr> %pv2, <16 x i32> <i32 poison, i32 21, i32 12, i32 poison, i32 29, i32 4, i32 poison, i32 poison, i32 27, i32 25, i32 poison, i32 poison, i32 20, i32 18, i32 poison, i32 8>
  %mask = freeze <16 x i1> poison
  call void @llvm.masked.scatter.v16i16.v16p0(<16 x i16> %call, <16 x ptr> align 2 %ptrs, <16 x i1> %mask)
  br label %common.ret
}
