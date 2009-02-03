; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=i686-pc-linux-gnu -relocation-model=pic -o %t -f 
; RUN: grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep piclabel %t | count 3
; RUN: grep PLT %t | count 1
; RUN: grep GOT %t | count 1
; RUN: not grep GOTOFF %t

@pfoo = external global void(...)* 

define void @bar() {
entry:
    %tmp = call void(...)*(...)* @afoo()
    store void(...)* %tmp, void(...)** @pfoo
    %tmp1 = load void(...)** @pfoo
    call void(...)* %tmp1()
    br label %return
return:
    ret void
}

declare void(...)* @afoo(...)
