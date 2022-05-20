#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdio.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

int *g_int_ptr = NULL;
size_t g_size = 10*sizeof(int);

void
saction_handler(int signo, siginfo_t info, void *baton) {
  printf("Got into handler.\n");   // stop here in the signal handler
  kern_return_t success
      = mach_vm_protect(mach_task_self(), g_int_ptr,
                        g_size, 0, VM_PROT_READ|VM_PROT_WRITE);
  g_int_ptr[1] = 20;
}
int
main()
{
  kern_return_t vm_result = vm_allocate(mach_task_self(), &g_int_ptr, g_size, VM_FLAGS_ANYWHERE);
  for (int i = 0; i < 10; i++)
    g_int_ptr[i] = i * 10;
  
  vm_result = mach_vm_protect(mach_task_self(), g_int_ptr, g_size, 0, VM_PROT_NONE);
  struct sigaction my_action;
  sigemptyset(&my_action.sa_mask);
  my_action.sa_handler = (void (*)(int)) saction_handler;
  my_action.sa_flags = SA_SIGINFO;

  sigaction(SIGBUS, &my_action, NULL); // Stop here to get things going.
  int local_value = g_int_ptr[1];
  return local_value; // Break here to make sure we got past the signal handler
}
