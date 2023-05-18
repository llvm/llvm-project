; RUN: llc %s -o - | FileCheck %s
; RUN: llc %s -o - | llvm-mc -triple=wasm32-unknown-unknown | FileCheck %s

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-wasi"

; Function Attrs: nounwind
define hidden i32 @d() local_unnamed_addr #0 {
entry:
  %0 = call i32 @g() #3
  call void @llvm.memset.p0.i32(ptr nonnull align 4 inttoptr (i32 4 to ptr), i8 0, i32 %0, i1 false)                                        ; preds = %for.body.preheader, %entry
  ret i32 undef
}

declare i32 @g(...) local_unnamed_addr #1

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg) #2

attributes #0 = { nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" }
attributes #1 = { "frame-pointer"="none" "no-prototype" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" }
attributes #2 = { argmemonly nofree nounwind willreturn writeonly }
attributes #3 = { nounwind }

; CHECK:         .functype       memset (i32, i32, i32) -> (i32)
; CHECK:         .functype       g () -> (i32)
; CHECK:         call    g
; CHECK:         call    memset
