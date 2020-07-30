struct things_to_sum {
    int a;
    int b;
    int c;
};

int sum_things(struct things_to_sum tts)
{
    return tts.a + tts.b + tts.c;
}

int main (int argc, char const *argv[])
{
    struct point_tag {
        int x;
        int y;
        char padding[0];
    }; //% self.expect("frame variable pt.padding[0]", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["pt.padding[0] = "])
       //% self.expect("frame variable pt.padding[1]", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["pt.padding[1] = "])
       //% self.expect_expr("pt.padding[0]", result_type="char")
       //% self.expect("image lookup -t point_tag", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ['padding[]'])

    struct {} empty;
    //% self.expect("frame variable empty", substrs = ["empty = {}"])
    //% self.expect("expression -- sizeof(empty)", substrs = ["= 0"])

    struct rect_tag {
        struct point_tag bottom_left;
        struct point_tag top_right;
    };
    struct point_tag pt = { 2, 3, {} };
    struct rect_tag rect = {{1, 2, {}}, {3, 4, {}}};
    struct things_to_sum tts = { 2, 3, 4 };

    int sum = sum_things(tts); //% self.expect_expr("&pt == (struct point_tag*)0", result_value="false")
                               //% self.expect_expr("sum_things(tts)", result_value="9")
    return 0;
}
