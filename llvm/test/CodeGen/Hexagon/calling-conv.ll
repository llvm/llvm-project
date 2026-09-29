; RUN: llc -mtriple=hexagon -mno-pairing -mno-compound <%s | FileCheck %s --check-prefix=CHECK-ONE
; RUN: llc -mtriple=hexagon -mno-pairing -mno-compound <%s | FileCheck %s --check-prefix=CHECK-TWO
; RUN: llc -mtriple=hexagon -mno-pairing -mno-compound <%s | FileCheck %s --check-prefix=CHECK-THREE

%s.0 = type { i32, i8, i64 }
%s.1 = type { i8, i64 }

@g0 = external global ptr

; CHECK-ONE:    memw(r29+#48) = r2
; CHECK-TWO:    memw(r29+#52) = r2
; CHECK-THREE:  memw(r29+#56) = r2

define void @f0(ptr noalias nocapture sret(%s.0) %a0, i32 %a1, i8 zeroext %a2, ptr byval(%s.0) nocapture readnone align 8 %a3, ptr byval(%s.1) nocapture readnone align 8 %a4) #0 {
b0:
  %v0 = alloca %s.0, align 8
  %v1 = load ptr, ptr @g0, align 4
  %v2 = sext i32 %a1 to i64
  %v3 = add nsw i64 %v2, 1
  %v4 = add nsw i32 %a1, 2
  %v5 = add nsw i64 %v2, 3
  call void @f1(ptr sret(%s.0) %v0, i32 45, ptr byval(%s.0) align 8 %v1, ptr byval(%s.0) align 8 %v1, i8 zeroext %a2, i64 %v3, i32 %v4, i64 %v5, i8 zeroext %a2, i8 zeroext %a2, i8 zeroext %a2, i32 45)
  store i32 20, ptr %v0, align 8
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %a0, ptr align 8 %v0, i32 16, i1 false)
  ret void
}

declare void @f1(ptr sret(%s.0), i32, ptr byval(%s.0) align 8, ptr byval(%s.0) align 8, i8 zeroext, i64, i32, i64, i8 zeroext, i8 zeroext, i8 zeroext, i32)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
