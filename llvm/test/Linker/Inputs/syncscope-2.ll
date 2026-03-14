define void @syncscope_2() {
  fence syncscope("image") seq_cst
  fence syncscope("agent") seq_cst
  fence syncscope("workgroup") seq_cst
  ret void
}
