; RUN: llc < %s -mtriple=thumb-apple-darwin \
; RUN:     -frame-pointer=all | not grep "r11"
; RUN: llc < %s -mtriple=thumb-linux-gnueabi \
; RUN:     -frame-pointer=all | not grep "r11"

define i32 @f() {
entry:
	ret i32 10
}
