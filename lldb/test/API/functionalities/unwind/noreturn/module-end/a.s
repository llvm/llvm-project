# compile this with:
# as a.s -o a.o --32 && ld a.o -m elf_i386
# generate core file with:
# ulimit -s 12 && ./a.out

.text

.globl func2
.type func2, @function
func2:
  pushl %ebp
  movl  %esp, %ebp
  movl  0,    %eax
  popl  %ebp
  ret
.size func2, .-func2

.globl _start
.type _start, @function
_start:
  pushl %ebp
  movl  %esp, %ebp
  call  func1
  popl  %ebp
  ret
.size _start, .-_start

.globl func1
.type func1, @function
func1:
  pushl %ebp
  movl  %esp, %ebp
  call  func2
.size func1, .-func1

