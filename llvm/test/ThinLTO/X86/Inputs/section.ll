; An internal global variable that can't be renamed because it has a section
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
@var_with_section = internal global i32 0, section "some_section"

; @reference_gv_with_section() can't be imported
define i32 @reference_gv_with_section() {
    %res = load i32, i32* @var_with_section
    ret i32 %res
}

; canary
define void @foo() {
    ret void
}
