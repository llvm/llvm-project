; RUN: llc < %s -mtriple=aarch64-pc-mingw32 | FileCheck %s -check-prefix=WINEH
; RUN: llc < %s -mtriple=aarch64-pc-mingw32 -filetype=obj | llvm-readobj -S - | FileCheck %s -check-prefix=WINEH-SECTIONS

; Check emission of eh handler and handler data
declare i32 @_d_eh_personality(i32, i32, i64, ptr, ptr)
declare void @_d_eh_resume_unwind(ptr)

declare i32 @bar()

define i32 @foo4() #0 personality ptr @_d_eh_personality {
entry:
  %step = alloca i32, align 4
  store i32 0, ptr %step
  %tmp = load i32, ptr %step

  %tmp1 = invoke i32 @bar()
          to label %finally unwind label %landingpad

finally:
  store i32 1, ptr %step
  br label %endtryfinally

landingpad:
  %landing_pad = landingpad { ptr, i32 }
          cleanup
  %tmp3 = extractvalue { ptr, i32 } %landing_pad, 0
  store i32 2, ptr %step
  call void @_d_eh_resume_unwind(ptr %tmp3)
  unreachable

endtryfinally:
  %tmp10 = load i32, ptr %step
  ret i32 %tmp10
}
; WINEH-LABEL: foo4:
; WINEH: .seh_proc foo4
; WINEH: .seh_handler _d_eh_personality, @unwind, @except
; WINEH: ret
; WINEH:      .seh_handlerdata
; WINEH-NEXT: .text
; WINEH-NEXT: .seh_endproc
; WINEH: .section .xdata,"dr"
; WINEH-NEXT: .p2align 2
; WINEH-NEXT: GCC_except_table0:

; WINEH-SECTIONS: Name: .xdata
; WINEH-SECTIONS-NOT: Name: .gcc_except_table
