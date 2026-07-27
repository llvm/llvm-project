$runtime_export = comdat any
$runtime_internal = comdat any

@runtime_export = linkonce_odr global i32 0, comdat, align 4
@runtime_internal = linkonce_odr global i32 0, comdat, align 4

define void @__runtime_export_pre_numeric(i32 %size, i32 %id) {
entry:
  %export = load i32, ptr @runtime_export, align 4
  %internal = load i32, ptr @runtime_internal, align 4
  %sum = add i32 %export, %internal
  store i32 %sum, ptr @runtime_internal, align 4
  ret void
}