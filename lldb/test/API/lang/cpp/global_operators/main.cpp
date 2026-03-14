#include <new>

struct new_tag_t
{
};
new_tag_t new_tag;

struct Struct {
	int value;
};

bool operator==(const Struct &a, const Struct &b) {
	return a.value == b.value;
}

typedef char buf_t[sizeof(Struct)];
buf_t global_new_buf, tagged_new_buf;

// This overrides global operator new
// This function and the following does not actually allocate memory. We are merely
// trying to make sure it is getting called.
void *
operator new(std::size_t count)
{
    return &global_new_buf;
}

// A custom allocator
void *
operator new(std::size_t count, const new_tag_t &)
{
    return &tagged_new_buf;
}

int main() {
	Struct s1, s2, s3;
	s1.value = 3;
	s2.value = 5;
	s3.value = 3;
	return 0; // break here
}

