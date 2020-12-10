; RUN: llc -march=hexagon -mno-pairing -mno-compound <%s | FileCheck %s --check-prefix=CHECK-ONE
; RUN: llc -march=hexagon -mno-pairing -mno-compound <%s | FileCheck %s --check-prefix=CHECK-TWO
; RUN: llc -march=hexagon -mno-pairing -mno-compound <%s | FileCheck %s --check-prefix=CHECK-THREE

%s.0 = type { i32, i8, i64 }
%s.1 = type { i8, i64 }

@g0 = external global %s.0*

; CHECK-ONE:    memw(r29+#48) = r2
; CHECK-TWO:    memw(r29+#52) = r2
; CHECK-THREE:  memw(r29+#56) = r2

define void @f0(%s.0* noalias nocapture sret(%s.0) %a0, i32 %a1, i8 zeroext %a2, %s.0* byval(%s.0) nocapture readnone align 8 %a3, %s.1* byval(%s.1) nocapture readnone align 8 %a4) #0 {
b0:
  %v0 = alloca %s.0, align 8
  %v1 = load %s.0*, %s.0** @g0, align 4
  %v2 = sext i32 %a1 to i64
  %v3 = add nsw i64 %v2, 1
  %v4 = add nsw i32 %a1, 2
  %v5 = add nsw i64 %v2, 3
  call void @f1(%s.0* sret(%s.0) %v0, i32 45, %s.0* byval(%s.0) align 8 %v1, %s.0* byval(%s.0) align 8 %v1, i8 zeroext %a2, i64 %v3, i32 %v4, i64 %v5, i8 zeroext %a2, i8 zeroext %a2, i8 zeroext %a2, i32 45)
  %v6 = bitcast %s.0* %v0 to i32*
  store i32 20, i32* %v6, align 8
  %v7 = bitcast %s.0* %a0 to i8*
  %v8 = bitcast %s.0* %v0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %v7, i8* align 8 %v8, i32 16, i1 false)
  ret void
}

declare void @f1(%s.0* sret(%s.0), i32, %s.0* byval(%s.0) align 8, %s.0* byval(%s.0) align 8, i8 zeroext, i64, i32, i64, i8 zeroext, i8 zeroext, i8 zeroext, i32)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { argmemonly nounwind }
