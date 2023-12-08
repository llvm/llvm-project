typedef struct t *tp;
typedef tp (*get_tp)();

struct s {
    get_tp get_tp_p;
};

struct t {
    struct s *s;
};

struct t t;
