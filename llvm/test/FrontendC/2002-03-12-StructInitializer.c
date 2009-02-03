// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* GCC was not emitting string constants of the correct length when
 * embedded into a structure field like this.  It thought the strlength
 * was -1.
 */

typedef struct Connection_Type {
   long    to;
   char    type[10];
   long    length;
} Connection;

Connection link[3]
= { {1, "link1", 10},
    {2, "link2", 20},
    {3, "link3", 30} };

