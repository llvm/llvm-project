; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -relocation-model=pic \
; RUN:   -o %t -f
; RUN: grep _GLOBAL_OFFSET_TABLE_ %t
; RUN: grep piclabel %t | count 3
; RUN: grep GOTOFF %t | count 4

@ptr = internal global i32* null
@dst = internal global i32 0
@src = internal global i32 0

define void @foo() {
entry:
    store i32* @dst, i32** @ptr
    %tmp.s = load i32* @src
    store i32 %tmp.s, i32* @dst
    ret void
}

