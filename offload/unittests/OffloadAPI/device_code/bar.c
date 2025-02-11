#include "util.h"

KERNEL void bar(int *out) {
    out[get_thread_id_x()] = get_thread_id_x() + 1;
}
