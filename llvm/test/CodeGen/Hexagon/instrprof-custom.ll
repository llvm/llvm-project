; RUN: llc -march=hexagon -relocation-model=pic < %s | FileCheck %s
; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test1:
; CHECK: {{call my_instrprof_handler|r0 = #999}}
; CHECK-NEXT: {{call my_instrprof_handler|r0 = #999}}

@handler_name = internal constant [21 x i8] c"my_instrprof_handler\00"

define dllexport void @test1() local_unnamed_addr #0 {
entry:
  tail call void @llvm.hexagon.instrprof.custom(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @handler_name, i32 0, i32 0), i32 999)
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.hexagon.instrprof.custom(i8*, i32) #1

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp,+hmxv68" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }
