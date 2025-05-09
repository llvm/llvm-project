
// This simple program is to test the lldb Python API SBTarget ReadInstruction
// function.
//
// When the target is create we get all the instructions using the intel
// flavor and see if it is correct.

int test_add(int a, int b);

__asm__("test_add:\n"
        "    movl    %edi, %eax\n"
        "    addl    %esi, %eax\n"
        "    ret     \n");

int main(int argc, char **argv) {
  int a = 10;
  int b = 20;
  int result = test_add(a, b);

  return 0;
}