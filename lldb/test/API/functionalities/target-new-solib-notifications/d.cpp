int d_init() { return 123; }

int d_global = d_init();

extern "C" int d_function() { return 700; }
