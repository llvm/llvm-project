#ifndef BRIDGING_HEADER
#define BRIDGING_HEADER
typedef struct TwoAnonymousStructs {
        struct {
            float x;
            float y;
            float z;
        };
        struct {
          int a;
        };
} TwoAnonymousStructs;

typedef struct TwoAnonymousUnions {
        union {
          struct {
            int x;
          };
          struct {
            int y;
            int z;
          };
        };
        union {
          struct {
            int a;
            int b;
            int c;
          };
          struct {
            int d;
            int e;
          };
        };
} TwoAnonymousUnions;

TwoAnonymousStructs makeTwoAnonymousStructs();
TwoAnonymousUnions makeTwoAnonymousUnions();
#endif
