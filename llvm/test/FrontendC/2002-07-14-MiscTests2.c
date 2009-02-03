// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


// Test ?: in function calls
extern fp(int, char*);
char *Ext;
void
__bb_exit_func (void)
{
  fp (12, Ext ? Ext : "<none>");
}


