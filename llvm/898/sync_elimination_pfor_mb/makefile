all: main_o3 main_sync

clean:
	rm -f main_o3 main_sync

main_o3:
	clang -O3 -o main_o3 main.c
	clang -O3 -S -emit-llvm -o main_o3.ll main.c

main_sync:
	clang -S -emit-llvm -o main_sync_raw.ll main.c
	opt -O3 main_sync_raw.ll -S -o main_sync_o3.ll
	opt -sync-elimination main_sync_o3.ll -S -o main_sync_sync_eliminated.ll
	opt -simplifycfg -loop-simplify -loop-fuse main_sync_sync_eliminated.ll -S -o main_sync_loop_fused.ll
	clang -O3 -o main_sync main_sync_loop_fused.ll
	clang -O3 -S -emit-llvm -o main_sync.ll main_sync_loop_fused.ll
