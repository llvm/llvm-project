// RUN: %llvmgcc -c %s  -o /dev/null
static char *str;

static const struct {
 const char *name;
 unsigned type;
} scan_special[] = {
 {"shift", 1},
 {0, 0}
};

static void
sb(void)
{
 while (*str == ' ' || *str == '\t')
  str++;
}
