// RUN: %llvmgcc %s -S -o -

void regnode(int op);

void regnode(op)
char op;
{
}
