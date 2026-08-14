@runtime_export = global i32 0, align 4
@runtime_internal = global i32 0, align 4

define void @__runtime_export_pre_numeric(i32 %size, i32 %id) {
entry:
  %export = load i32, ptr @runtime_export, align 4
  %internal = load i32, ptr @runtime_internal, align 4
  %sum = add i32 %export, %internal
  store i32 %sum, ptr @runtime_internal, align 4
  ret void
}
