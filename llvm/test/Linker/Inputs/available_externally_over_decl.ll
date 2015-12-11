@h = global void ()* @f
@h2 = global void ()* @g

define available_externally void @f() {
  ret void
}

define available_externally void @g() {
  ret void
}
