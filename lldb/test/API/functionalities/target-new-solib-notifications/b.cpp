int b_init() { return 345; }

int b_global = b_init();

extern "C" int b_function() { return 500; }
