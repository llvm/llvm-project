; How to make the shared objects from this file:
;
; LDARGS="--unresolved-symbols=ignore-all -soname=libfoo.so --no-as-needed -lc -lm"
;
; X86-32 ELF:
;   llc -mtriple=i386-linux-gnu shared.ll -filetype=obj -o tmp32.o -relocation-model=pic
;   ld -melf_i386 -shared tmp32.o -o shared-object-test.elf-i386 $LDARGS
;
; X86-64 ELF:
;   llc -mtriple=x86_64-linux-gnu shared.ll -filetype=obj -o tmp64.o -relocation-model=pic
;   ld -melf_x86_64 -shared tmp64.o -o shared-object-test.elf-x86-64 $LDARGS

@defined_sym = global i32 1, align 4

@tls_sym = thread_local global i32 2, align 4

@undef_sym = external global i32

@undef_tls_sym = external thread_local global i32

@common_sym = common global i32 0, align 4

define i32 @global_func() nounwind uwtable {
entry:
  ret i32 0
}

declare i32 @undef_func(...)

define internal i32 @local_func() nounwind uwtable {
entry:
  ret i32 0
}
