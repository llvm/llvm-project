;RUN: llc -march=sparc   < %s | FileCheck %s
;RUN: llc -march=sparcv9 < %s | FileCheck %s --check-prefix=V9


%0 = type { [32 x i32] }
%struct.jmpbuf_env = type { i32, i32, [1 x %struct.__jmp_buf_tag], i32 }
%struct.__jmp_buf_tag = type { [3 x i32], i32, %0 }

@jenv = common unnamed_addr global %struct.jmpbuf_env* null
@.cst = private unnamed_addr constant [30 x i8] c"in bar with jmp_buf's id: %d\0A\00", align 64

; CHECK-LABEL: foo
; CHECK-DAG:   st {{.+}}, [%i0]
; CHECK-DAG:   st {{.+}}, [%i0+4]
; CHECK:       call _setjmp
; CHECK:       ld [%fp+{{.+}}], %[[R:[gilo][0-7]]]
; CHECK:       st %o0, [%[[R]]+{{.+}}]

; V9-LABEL:   foo
; V9-DAG:     st {{.+}}, [%i0]
; V9-DAG:     st {{.+}}, [%i0+4]
; V9:         call _setjmp
; V9:         ldx [%fp+{{.+}}], %[[R:[gilo][0-7]]]
; V9:         st %o0, [%[[R]]+{{.+}}]

; Function Attrs: nounwind
define i32 @foo(%struct.jmpbuf_env* byval(%struct.jmpbuf_env) %inbuf) #0 {
entry:
  %0 = getelementptr inbounds %struct.jmpbuf_env, %struct.jmpbuf_env* %inbuf, i32 0, i32 0
  store i32 0, i32* %0, align 4, !tbaa !4
  %1 = getelementptr inbounds %struct.jmpbuf_env, %struct.jmpbuf_env* %inbuf, i32 0, i32 1
  store i32 1, i32* %1, align 4, !tbaa !4
  %2 = getelementptr inbounds %struct.jmpbuf_env, %struct.jmpbuf_env* %inbuf, i32 0, i32 2, i32 0
  %3 = call i32 @_setjmp(%struct.__jmp_buf_tag* %2) #2
  %4 = getelementptr inbounds %struct.jmpbuf_env, %struct.jmpbuf_env* %inbuf, i32 0, i32 3
  store i32 %3, i32* %4, align 4, !tbaa !4
  store %struct.jmpbuf_env* %inbuf, %struct.jmpbuf_env** @jenv, align 4, !tbaa !3
  %5 = load i32, i32* %1, align 4, !tbaa !4
  %6 = icmp eq i32 %5, 1
  %7 = icmp eq i32 %3, 0
  %or.cond = and i1 %6, %7
  br i1 %or.cond, label %"4.i", label %bar.exit

"4.i":                                            ; preds = %entry
  call void @longjmp(%struct.__jmp_buf_tag* %2, i32 0) #1
  unreachable

bar.exit:                                         ; preds = %entry
  %8 = load i32, i32* %0, align 4, !tbaa !4
  %9 = call i32 (i8*, ...) @printf(i8* noalias getelementptr inbounds ([30 x i8], [30 x i8]* @.cst, i32 0, i32 0), i32 %8) #0
  ret i32 0
}

; Function Attrs: nounwind returns_twice
declare i32 @_setjmp(%struct.__jmp_buf_tag*) #2

; Function Attrs: noreturn nounwind
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture, ...) #0


attributes #0 = { nounwind }
attributes #1 = { noreturn nounwind }
attributes #2 = { nounwind returns_twice }

!0 = !{!"alias set 6: struct.jmpbuf_env*", !1}
!1 = !{!1}
!2 = !{!"alias set 3: int", !1}
!3 = !{!0, !0, i64 0}
!4 = !{!2, !2, i64 0}
