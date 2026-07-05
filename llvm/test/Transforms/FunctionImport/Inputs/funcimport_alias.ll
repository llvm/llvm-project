declare void @analias()

define void @callanalias() #0 {
entry:
  call void @analias()
  ret void
}
