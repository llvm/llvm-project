target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare i8 **@foo()
define i32 @main() {
    call i8 **@foo()
	ret i32 0
}
