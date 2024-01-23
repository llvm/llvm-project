// A script to (re)create the .yaml files is in 'make-inputs'. If you make changes
// you'll need to update the .note.gnu.buildid values in the tests, as the cache names

int func(int argc, const char **argv) { return (argc + 1) * (argv[argc][0] + 2); }

__attribute__((force_align_arg_pointer)) void _start(void) {

  /* main body of program: call main(), etc */

  const char *argv[] = {""};
  func(0, argv);

  /* exit system call */
  asm("mov $60,%rax; mov $0,%rdi; syscall");
  __builtin_unreachable(); // tell the compiler to make sure side effects are done before the asm statement
}
