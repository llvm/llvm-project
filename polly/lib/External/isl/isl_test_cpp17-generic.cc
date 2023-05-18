/* A class that sets a boolean when an object of the class gets destroyed.
 */
struct S {
	S(bool *freed) : freed(freed) {}
	~S();

	bool *freed;
};

/* S destructor.
 *
 * Set the boolean, a pointer to which was passed to the constructor.
 */
S::~S()
{
	*freed = true;
}

/* Construct an isl::id with an S object attached that sets *freed
 * when it gets destroyed.
 */
static isl::id construct_id(isl::ctx ctx, bool *freed)
{
	auto s = std::make_shared<S>(freed);
	isl::id id(ctx, "S", s);
	return id;
}

/* Test id::try_user.
 *
 * In particular, check that the object attached to an identifier
 * can be retrieved again, that trying to retrieve an object of the wrong type
 * or trying to retrieve an object when no object was attached fails.
 * Furthermore, check that the object attached to an identifier
 * gets properly freed.
 */
static void test_try_user(isl::ctx ctx)
{
	isl::id id(ctx, "test", 5);
	isl::id id2(ctx, "test2");

	auto maybe_int = id.try_user<int>();
	auto maybe_s = id.try_user<std::shared_ptr<S>>();
	auto maybe_int2 = id2.try_user<int>();

	if (!maybe_int)
		die("integer cannot be retrieved from isl::id");
	if (maybe_int.value() != 5)
		die("wrong integer retrieved from isl::id");
	if (maybe_s)
		die("structure unexpectedly retrieved from isl::id");
	if (maybe_int2)
		die("integer unexpectedly retrieved from isl::id");

	bool freed = false;
	{
		isl::id id = construct_id(ctx, &freed);
		if (freed)
			die("data structure freed prematurely");
		auto maybe_s = id.try_user<std::shared_ptr<S>>();
		if (!maybe_s)
			die("structure cannot be retrieved from isl::id");
		if (maybe_s.value()->freed != &freed)
			die("invalid structure retrieved from isl::id");
	}
	if (!freed)
		die("data structure not freed");
}
