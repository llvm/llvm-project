; RUN: opt < %s -passes=internalize -internalize-public-api-list main -S | FileCheck %s

@A = global i32 0
; CHECK: @A = internal global i32 0

@B = alias i32, ptr @A
; CHECK: @B = internal alias i32, ptr @A

@C = alias i32, ptr @A
; CHECK: @C = internal alias i32, ptr @A

define i32 @main() {
	%tmp = load i32, ptr @C
	ret i32 %tmp
}

; CHECK: define i32 @main() {
