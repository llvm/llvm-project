define void @g() {
entry:
  ret void
}

@analias = alias void (...), bitcast (void ()* @aliasee to void (...)*)
define void @aliasee() {
entry:
  ret void
}
