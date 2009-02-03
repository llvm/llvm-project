#define TRUE_VAL (!0)
#define FALSE_VAL 0
#define ARR_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

typedef unsigned long long int uint64_t;
typedef long long int int64_t;

/* ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- */
struct harness_int64_pred {
  const char     *fmt_string;
  int64_t        *lhs;
  int64_t        *rhs;
  int64_t        *select_a;
  int64_t        *select_b;
  int             expected;
  int64_t        *select_expected;
};

struct harness_uint64_pred {
  const char     *fmt_string;
  uint64_t       *lhs;
  uint64_t       *rhs;
  uint64_t       *select_a;
  uint64_t       *select_b;
  int             expected;
  uint64_t       *select_expected;
};

struct int64_pred_s {
  const char     *name;
  int             (*predfunc) (int64_t, int64_t);
  int64_t         (*selfunc) (int64_t, int64_t, int64_t, int64_t);
  struct harness_int64_pred *tests;
  int             n_tests;
};

struct uint64_pred_s {
  const char     *name;
  int             (*predfunc) (uint64_t, uint64_t);
  uint64_t        (*selfunc) (uint64_t, uint64_t, uint64_t, uint64_t);
  struct harness_uint64_pred *tests;
  int             n_tests;
};
