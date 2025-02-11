#include "util.h"

KERNEL void foo(int *out) {
    out[get_thread_id_x()] = get_thread_id_x();
}
