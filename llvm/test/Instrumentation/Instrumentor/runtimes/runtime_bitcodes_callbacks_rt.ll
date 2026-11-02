@runtime_state = external global i32
@runtime_private_state = internal global i32 0, align 4

define protected void @__runtime_bitcodes_pre_numeric(i32 %size, i32 %id) {
entry:
  %state = load i32, ptr @runtime_state, align 4
  %private_state = load i32, ptr @runtime_private_state, align 4
  %sum = add i32 %state, %private_state
  store i32 %sum, ptr @runtime_private_state, align 4
  ret void
}
