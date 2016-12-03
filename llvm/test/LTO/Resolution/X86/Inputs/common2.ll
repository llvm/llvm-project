target triple = "x86_64-apple-macosx10.11.0"

@v = common global i16 0, align 4

define i16 *@bar() {
 ret i16 *@v
}