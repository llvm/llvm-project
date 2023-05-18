define void @syncscope_1() {
  fence syncscope("agent") seq_cst
  fence syncscope("workgroup") seq_cst
  fence syncscope("wavefront") seq_cst
  ret void
}
