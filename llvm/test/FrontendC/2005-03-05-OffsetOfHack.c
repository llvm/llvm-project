// RUN: %llvmgcc %s -S -o - 

struct s {
  unsigned long int field[0];
};

#define OFFS \
        (((char *) &((struct s *) 0)->field[0]) - (char *) 0)

int foo[OFFS];


