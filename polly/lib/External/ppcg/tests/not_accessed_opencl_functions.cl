void copy(__global int b[1000], __global int a[1000], int pos,
	__global int c[1000])
{
	b[pos] = a[pos];
}
