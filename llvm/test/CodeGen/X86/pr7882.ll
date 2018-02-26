; RUN: llc < %s -mtriple=i686-apple-darwin -pre-RA-sched=fast \
; RUN: | FileCheck %s
; make sure scheduler honors the flags clobber.  PR 7882.

define i32 @main(i32 %argc, i8** %argv) nounwind
{
entry:
; CHECK: InlineAsm End
; CHECK: cmpl
    %res = icmp slt i32 1, %argc
    %tmp = call i32 asm sideeffect alignstack
        "push $$0
         popf
         mov $$13, $0", "=r,r,~{memory},~{flags}" (i1 %res)
    %ret = select i1 %res, i32 %tmp, i32 42
    ret i32 %ret
}
