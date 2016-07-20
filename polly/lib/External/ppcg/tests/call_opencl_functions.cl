void copy(__global int b[1000], __global int a[1000], int pos)
{
	b[pos] = a[pos];
}
