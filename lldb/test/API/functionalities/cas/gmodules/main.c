int cached_fn(void);
int uncached_fn(void);

int main(void) { return cached_fn() + uncached_fn() - 58; }
