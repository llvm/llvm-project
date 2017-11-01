; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s -check-prefix=LINUX -check-prefix=CHECK
; RUN: llc < %s -mtriple=i686-apple-darwin | FileCheck %s -check-prefix=DARWIN -check-prefix=CHECK

declare i32 @__gxx_personality_v0(...)
declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)
declare void @large(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f)
declare void @empty()

; When we use an invoke, we expect a .cfi_escape GNU_ARGS_SIZE
; with size 16 before the invocation. Without FP, we also expect
; .cfi_adjust_cfa_offset after each push.
; Darwin should not generate pushes in either circumstance.
; CHECK-LABEL: test1_nofp:
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX-NEXT: pushl   $4
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $3
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $2
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $1
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: call
; LINUX-NEXT: addl $16, %esp
; LINUX: .cfi_adjust_cfa_offset -16
; DARWIN-NOT: .cfi_escape
; DARWIN-NOT: pushl
define void @test1_nofp() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; CHECK-LABEL: test1_fp:
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX-NEXT: pushl   $4
; LINUX-NEXT: pushl   $3
; LINUX-NEXT: pushl   $2
; LINUX-NEXT: pushl   $1
; LINUX-NEXT: call
; LINUX-NEXT: addl $16, %esp
; DARWIN: pushl %ebp
; DARWIN-NOT: .cfi_escape
; DARWIN-NOT: pushl
define void @test1_fp() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; If the function has no handlers, we don't need to generate GNU_ARGS_SIZE,
; even if it has an unwind table. Without FP, we still need cfi_adjust_cfa_offset,
; so darwin should not generate pushes.
; CHECK-LABEL: test2_nofp:
; LINUX-NOT: .cfi_escape
; LINUX: pushl   $4
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $3
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $2
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $1
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: call
; LINUX-NEXT: addl $16, %esp
; LINUX: .cfi_adjust_cfa_offset -16
; LINUX: addl $12, %esp
; DARWIN-NOT: .cfi_escape
; DARWIN-NOT: pushl
define void @test2_nofp() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; CHECK-LABEL: test2_fp:
; CHECK-NOT: .cfi_escape
; CHECK-NOT: .cfi_adjust_cfa_offset
; CHECK: pushl   $4
; CHECK-NEXT: pushl   $3
; CHECK-NEXT: pushl   $2
; CHECK-NEXT: pushl   $1
; CHECK-NEXT: call
; CHECK-NEXT: addl $24, %esp
define void @test2_fp() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; If we did not end up using any pushes, no need for GNU_ARGS_SIZE or
; cfi_adjust_cfa_offset.
; CHECK-LABEL: test3_nofp:
; LINUX-NOT: .cfi_escape
; LINUX-NOT: .cfi_adjust_cfa_offset
; LINUX-NOT: pushl
; LINUX: retl
define void @test3_nofp() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @empty()
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; If we did not end up using any pushes, no need for GNU_ARGS_SIZE or
; cfi_adjust_cfa_offset.
; CHECK-LABEL: test3_fp:
; LINUX: pushl %ebp
; LINUX-NOT: .cfi_escape
; LINUX-NOT: .cfi_adjust_cfa_offset
; LINUX-NOT: pushl
; LINUX: retl
define void @test3_fp() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @empty()
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; Different sized stacks need different GNU_ARGS_SIZEs
; CHECK-LABEL: test4:
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX-NEXT: pushl   $4
; LINUX-NEXT: pushl   $3
; LINUX-NEXT: pushl   $2
; LINUX-NEXT: pushl   $1
; LINUX-NEXT: call
; LINUX-NEXT: addl $16, %esp
; LINUX: .cfi_escape 0x2e, 0x20
; LINUX: subl    $8, %esp
; LINUX-NEXT: pushl   $11
; LINUX-NEXT: pushl   $10
; LINUX-NEXT: pushl   $9
; LINUX-NEXT: pushl   $8
; LINUX-NEXT: pushl   $7
; LINUX-NEXT: pushl   $6
; LINUX-NEXT: calll   large
; LINUX-NEXT: addl $32, %esp
define void @test4() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @large(i32 6, i32 7, i32 8, i32 9, i32 10, i32 11)
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; If we did use pushes, we need to reset GNU_ARGS_SIZE before a call
; without parameters, but don't need to adjust the cfa offset
; CHECK-LABEL: test5_nofp:
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX-NEXT: pushl   $4
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $3
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $2
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: pushl   $1
; LINUX-NEXT: .cfi_adjust_cfa_offset 4
; LINUX-NEXT: call
; LINUX-NEXT: addl $16, %esp
; LINUX: .cfi_adjust_cfa_offset -16
; LINUX-NOT: .cfi_adjust_cfa_offset
; LINUX: .cfi_escape 0x2e, 0x00
; LINUX-NOT: .cfi_adjust_cfa_offset
; LINUX: call
define void @test5_nofp() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @empty()
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; CHECK-LABEL: test5_fp:
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX-NEXT: pushl   $4
; LINUX-NEXT: pushl   $3
; LINUX-NEXT: pushl   $2
; LINUX-NEXT: pushl   $1
; LINUX-NEXT: call
; LINUX-NEXT: addl $16, %esp
; LINUX: .cfi_escape 0x2e, 0x00
; LINUX-NOT: .cfi_adjust_cfa_offset
; LINUX: call
define void @test5_fp() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @empty()
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; FIXME: This is actually inefficient - we don't need to repeat the .cfi_escape twice.
; CHECK-LABEL: test6:
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX: call
; LINUX: .cfi_escape 0x2e, 0x10
; LINUX: call
define void @test6() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue1 unwind label %cleanup
continue1:
  invoke void @good(i32 5, i32 6, i32 7, i32 8)
          to label %continue2 unwind label %cleanup
continue2:
  ret void          
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

; Darwin should generate pushes in the presense of FP and an unwind table,
; but not FP and invoke.
; CHECK-LABEL: test7:
; DARWIN: pushl %ebp
; DARWIN: movl %esp, %ebp
; DARWIN: .cfi_def_cfa_register %ebp
; DARWIN-NOT: .cfi_adjust_cfa_offset
; DARWIN: pushl   $4
; DARWIN-NEXT: pushl   $3
; DARWIN-NEXT: pushl   $2
; DARWIN-NEXT: pushl   $1
; DARWIN-NEXT: call
define void @test7() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  call void @good(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; CHECK-LABEL: test8:
; DARWIN: pushl %ebp
; DARWIN: movl %esp, %ebp
; DARWIN-NOT: .cfi_adjust_cfa_offset
; DARWIN-NOT: pushl
define void @test8() #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:  
  landingpad { i8*, i32 }
     cleanup
  ret void
}

attributes #0 = { optsize }
attributes #1 = { optsize "no-frame-pointer-elim"="true" }
