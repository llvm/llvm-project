typedef struct t *tp;
typedef tp (*get_tp)();

struct t {
    struct {
      get_tp get_tp_p;
    };
};

struct t t;
