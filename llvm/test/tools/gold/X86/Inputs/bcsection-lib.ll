declare void @elf_func()

define i32 @lib_func() {
  call void @elf_func()
  ret i32 0
}
