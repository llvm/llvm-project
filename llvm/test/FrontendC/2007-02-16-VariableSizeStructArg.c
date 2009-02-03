// RUN: %llvmgcc -S -w %s -o - 
// PR1170
int f(int a, struct {int b[a];} c) {  return c.b[0]; }

int g(struct {int b[1];} c) {
  return c.b[0];
}
