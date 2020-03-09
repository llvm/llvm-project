#include <stdio.h>
#include <stdint.h>

struct RegisterContext
{
    uintptr_t r0;
    uintptr_t r1;
    uintptr_t r2;
    uintptr_t r3;
    uintptr_t r4;
    uintptr_t pc;
    uintptr_t fp;
    uintptr_t sp;
};

struct ThreadInfo {
    uint32_t tid;
    const char *name;
    RegisterContext regs;
    ThreadInfo *next;
};
int main (int argc, char const *argv[], char const *envp[]);

ThreadInfo g_thread2 = { 0x2222, "thread2", { 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, (uintptr_t)&main, 0x2006, 0x2007 }, NULL       };
ThreadInfo g_thread1 = { 0x1111, "thread1", { 0x1000, 0x1001, 0x1002, 0x1003, 0x1004, (uintptr_t)&main, 0x1006, 0x1007 }, &g_thread2 };
ThreadInfo *g_thread_list_ptr = &g_thread1;

int main (int argc, char const *argv[], char const *envp[])
{
    printf ("g_thread_list is %p\n", g_thread_list_ptr);
    return 0; //% v = self.dbg.GetSelectedTarget().FindFirstGlobalVariable('g_thread_list_ptr')
    //% v_gla = v.GetChildMemberWithName('regs').GetLoadAddress()
    //% v_aof = v.GetChildMemberWithName('regs').AddressOf().GetValueAsUnsigned(lldb.LLDB_INVALID_ADDRESS)
    //% expr = '(%s)0x%x' % (v.GetType().GetName(), v.GetValueAsUnsigned(0))
    //% e = v.CreateValueFromExpression('e', expr)
    //% e_gla = e.GetChildMemberWithName('regs').GetLoadAddress()
    //% e_aof = e.GetChildMemberWithName('regs').AddressOf().GetValueAsUnsigned(lldb.LLDB_INVALID_ADDRESS)
    //% self.assertTrue(v_gla == e_gla, "GetLoadAddress() differs")
    //% self.assertTrue(v_aof == e_aof, "AddressOf() differs")
}
