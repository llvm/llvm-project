set terminal png

##------- Plot small Date vs LOC ----
set output "running_loc.png"
set xlabel "Date" 
set ylabel "Lines of Code"
set xdata time
set timefmt "%Y-%m-%d-%H:%M:%S:"
set format x "%b %d, %Y"

set size .75,.75
set xtics rotate
set xlabel 0,-1
plot "running_loc.txt" using 1:2 title '' with lines, \
     "running_loc.txt" using 1:2 title "Date vs. Lines of Code" with lines

##------- Plot large Date vs LOC ----
set size 1.5,1.5
set xtics norotate
set xlabel 0,0
set output "running_loc_large.png"
plot "running_loc.txt" using 1:2 title '', \
     "running_loc.txt" using 1:2 title "Date vs. Lines of Code" with lines


# Delete all labels...
set nolabel

##------- Olden CBE performance ----

set size .75,.75
set xtics rotate
set xlabel 0,-1
set output "running_Olden_cbe_time.png"
set ylabel "CBE compiled execution time (s)"
plot "running_Olden_cbe_time.txt" u 1:2 t '' with lines, \
     "running_Olden_cbe_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_cbe_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_cbe_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_cbe_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_cbe_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_cbe_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_cbe_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_cbe_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_cbe_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_cbe_time.txt" u 1:11 t "voronoi" \
   with lines

set size 1.5,1.5
set xtics norotate
set xlabel 0,0
set output "running_Olden_cbe_time_large.png"
plot "running_Olden_cbe_time.txt" u 1:2 t '' with lines, \
     "running_Olden_cbe_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_cbe_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_cbe_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_cbe_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_cbe_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_cbe_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_cbe_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_cbe_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_cbe_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_cbe_time.txt" u 1:11 t "voronoi" \
   with lines

##------- Olden JIT performance ----

set size .75,.75
set xtics rotate
set xlabel 0,-1
set output "running_Olden_jit_time.png"
set ylabel "JIT execution time (s)"
plot "running_Olden_jit_time.txt" u 1:2 t '' with lines, \
     "running_Olden_jit_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_jit_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_jit_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_jit_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_jit_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_jit_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_jit_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_jit_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_jit_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_jit_time.txt" u 1:11 t "voronoi" \
   with lines

set size 1.5,1.5
set xtics norotate
set xlabel 0,0
set output "running_Olden_jit_time_large.png"
plot "running_Olden_jit_time.txt" u 1:2 t '' with lines, \
     "running_Olden_jit_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_jit_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_jit_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_jit_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_jit_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_jit_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_jit_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_jit_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_jit_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_jit_time.txt" u 1:11 t "voronoi" \
   with lines

##------- Olden LLC performance ----

set size .75,.75
set xtics rotate
set xlabel 0,-1
set output "running_Olden_llc_time.png"
set ylabel "LLC compiled execution time (s)"
plot "running_Olden_llc_time.txt" u 1:2 t '' with lines, \
     "running_Olden_llc_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_llc_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_llc_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_llc_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_llc_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_llc_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_llc_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_llc_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_llc_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_llc_time.txt" u 1:11 t "voronoi" \
   with lines

set size 1.5,1.5
set xtics norotate
set xlabel 0,0
set output "running_Olden_llc_time_large.png"
plot "running_Olden_llc_time.txt" u 1:2 t '' with lines, \
     "running_Olden_llc_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_llc_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_llc_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_llc_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_llc_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_llc_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_llc_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_llc_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_llc_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_llc_time.txt" u 1:11 t "voronoi" \
   with lines


##------- Olden optimizer time ----

set size .75,.75
set xtics rotate
set xlabel 0,-1
set output "running_Olden_opt_time.png"
set ylabel "Time to run the optimizer (s)"
plot "running_Olden_opt_time.txt" u 1:2 t '' with lines, \
     "running_Olden_opt_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_opt_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_opt_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_opt_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_opt_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_opt_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_opt_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_opt_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_opt_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_opt_time.txt" u 1:11 t "voronoi" \
   with lines

set size 1.5,1.5
set xtics norotate
set xlabel 0,0
set output "running_Olden_opt_time_large.png"
plot "running_Olden_opt_time.txt" u 1:2 t '' with lines, \
     "running_Olden_opt_time.txt" u 1:2 t "bh" with lines, \
     "running_Olden_opt_time.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_opt_time.txt" u 1:4 t "mst" with lines, \
     "running_Olden_opt_time.txt" u 1:5 t "power" with lines, \
     "running_Olden_opt_time.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_opt_time.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_opt_time.txt" u 1:8 t "health" with lines, \
     "running_Olden_opt_time.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_opt_time.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_opt_time.txt" u 1:11 t "voronoi" \
   with lines


##------- Bytecode size ----

set size .75,.75
set xtics rotate
set xlabel 0,-1
set output "running_Olden_bytecode.png"
set ylabel "Program bytecode size (bytes)"
plot "running_Olden_bytecode.txt" u 1:2 t '' with lines, \
     "running_Olden_bytecode.txt" u 1:2 t "bh" with lines, \
     "running_Olden_bytecode.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_bytecode.txt" u 1:4 t "mst" with lines, \
     "running_Olden_bytecode.txt" u 1:5 t "power" with lines, \
     "running_Olden_bytecode.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_bytecode.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_bytecode.txt" u 1:8 t "health" with lines, \
     "running_Olden_bytecode.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_bytecode.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_bytecode.txt" u 1:11 t "voronoi" \
   with lines

set size 1.5,1.5
set xtics norotate
set xlabel 0,0
set output "running_Olden_bytecode_large.png"
plot "running_Olden_bytecode.txt" u 1:2 t '' with lines, \
     "running_Olden_bytecode.txt" u 1:2 t "bh" with lines, \
     "running_Olden_bytecode.txt" u 1:3 t "em3d" with lines, \
     "running_Olden_bytecode.txt" u 1:4 t "mst" with lines, \
     "running_Olden_bytecode.txt" u 1:5 t "power" with lines, \
     "running_Olden_bytecode.txt" u 1:6 t "tsp" with lines, \
     "running_Olden_bytecode.txt" u 1:7 t "bisort" with lines, \
     "running_Olden_bytecode.txt" u 1:8 t "health" with lines, \
     "running_Olden_bytecode.txt" u 1:9 t "perimeter" with lines, \
     "running_Olden_bytecode.txt" u 1:10 t "treeadd" with lines, \
     "running_Olden_bytecode.txt" u 1:11 t "voronoi" \
   with lines
