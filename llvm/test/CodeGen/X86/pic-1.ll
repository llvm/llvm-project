; RUN: llvm-as < %s | \
; RUN:   llc -mtriple=i686-pc-linux-gnu -relocation-model=pic -o %t -f
; RUN: grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep piclabel %t | count 3
; RUN: grep GOT %t | count 3
; RUN: not grep GOTOFF %t

@ptr = external global i32* 
@dst = external global i32 
@src = external global i32 

define void @foo() {
entry:
    store i32* @dst, i32** @ptr
    %tmp.s = load i32* @src
    store i32 %tmp.s, i32* @dst
    ret void
}

