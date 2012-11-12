; RUN: opt < %s -internalize -internalize-public-api-list main -S | grep internal | count 3

@A = global i32 0
@B = alias i32* @A
@C = alias i32* @B

define i32 @main() {
	%tmp = load i32* @C
	ret i32 %tmp
}
