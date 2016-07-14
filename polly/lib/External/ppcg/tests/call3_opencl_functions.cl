void copy(__global int b[100], __global int a[100])
{
	for (int i = 0; i < 100; ++i)
		b[i] = a[i];
}
