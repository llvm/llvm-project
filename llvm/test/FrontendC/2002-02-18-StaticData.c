// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


double FOO = 17;
double BAR = 12.0;
float XX = 12.0f;

static char *procnames[] = {
  "EXIT"
};

void *Data[] = { &FOO, &BAR, &XX };

