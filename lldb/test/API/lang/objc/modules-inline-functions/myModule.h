int notInline();

static __inline__ __attribute__ ((always_inline)) int isInline(int a)
{
    int b = a + a;
    return b;
}
