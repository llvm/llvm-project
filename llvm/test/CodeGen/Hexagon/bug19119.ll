; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: .sdata.4.g0,"aM"

target triple = "hexagon-unknown--elf"

%s.0 = type { i32 }

@g0 = global %s.0 { i32 3 }, align 4 #0
@g1 = global i32 0, align 4 #1
@g2 = global ptr @g0, align 4 #2
@g3 = global i32 0, align 4 #3
@g4 = global i32 0, align 4 #4

; Function Attrs: nounwind optsize
define i32 @f0() #5 section ".text.main" {
b0:
  %v0 = load i32, ptr @g3, align 4, !tbaa !4
  %v1 = add nsw i32 %v0, 1
  store i32 %v1, ptr @g3, align 4, !tbaa !4
  %v2 = load ptr, ptr @g2, align 4, !tbaa !8
  %v3 = load i32, ptr @g1, align 4, !tbaa !10
  %v4 = getelementptr inbounds i8, ptr %v2, i32 %v3
  %v6 = load i32, ptr %v4, align 4, !tbaa !4
  store i32 %v6, ptr @g4, align 4, !tbaa !4
  store i32 1, ptr @g3, align 4, !tbaa !4
  ret i32 0
}

attributes #0 = { "linker_input_section"=".sdata.4.cccc" "linker_output_section"=".sdata.4" }
attributes #1 = { "linker_input_section"=".sbss.4.np" "linker_output_section"=".sbss.4" }
attributes #2 = { "linker_input_section"=".sdata.4.cp" "linker_output_section"=".sdata.4" }
attributes #3 = { "linker_input_section"=".sbss.4.counter" "linker_output_section"=".sbss.4" }
attributes #4 = { "linker_input_section"=".sbss.4.value" "linker_output_section"=".sbss.4" }
attributes #5 = { nounwind optsize "target-cpu"="hexagonv55" }

!llvm.module.flags = !{!0, !2}

!0 = !{i32 6, !"Target CPU", !1}
!1 = !{!"hexagonv55"}
!2 = !{i32 6, !"Target Features", !3}
!3 = !{!"-hvx"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !6, i64 0}
!10 = !{!6, !6, i64 0}
