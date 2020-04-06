int main() {
#if defined(__x86_64__)
  // We setup two fake frames with frame pointer linking. The test will then
  // check that lldb-server's jThreadsInfo reply includes the top frame's
  // contents and the linked list of (frame-pointer, return-address) pairs. We
  // pretend the next frame is too large to stop the frame walk.
  asm volatile("movabsq $0xf00d, %rax\n\t"
               "pushq   %rax\n\t"               // fake return address
               "leaq    0x1000(%rsp), %rbp\n\t" // larger than kMaxFrameSize
               "pushq   %rbp\n\t"
               "movq    %rsp, %rbp\n\t"
               "pushq   $1\n\t" // fake frame contents
               "pushq   $2\n\t"
               "pushq   $3\n\t"
               "\n\t"
               "movabsq $0xfeed, %rax\n\t"
               "push    %rax\n\t" // second fake return address
               "pushq   %rbp\n\t"
               "movq    %rsp, %rbp\n\t"
               "pushq   $4\n\t" // fake frame contents
               "pushq   $5\n\t"
               "pushq   $6\n\t"
               "\n\t"
               "int3\n\t");
#endif
  return 0;
}
